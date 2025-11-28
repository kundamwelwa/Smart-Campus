"""
Academic Service gRPC Server

Minimal gRPC server exposing core academic operations using the compiled
protos in `shared.grpc.generated`.

This runs alongside the FastAPI HTTP service and uses the same domain logic.
"""

import asyncio

import grpc
from shared.grpc.generated import academic_pb2 as academic_pb2
from shared.grpc.generated import academic_pb2_grpc as academic_pb2_grpc

from shared.config import settings


class AcademicServiceGRPC(academic_pb2_grpc.AcademicServiceServicer):
  async def GetCourse(self, request: academic_pb2.GetCourseRequest, context) -> academic_pb2.CourseResponse:  # type: ignore[override]
    # For brevity, this is a thin wrapper over the HTTP API; in a full implementation
    # it would call the same application/service layer used by FastAPI.
    # TODO: Integrate directly with Academic domain/service layer.
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details("GetCourse via gRPC not fully implemented yet.")
    return academic_pb2.CourseResponse()

  async def ListCourses(self, request: academic_pb2.ListCoursesRequest, context) -> academic_pb2.CourseListResponse:  # type: ignore[override]
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details("ListCourses via gRPC not fully implemented yet.")
    return academic_pb2.CourseListResponse()

  async def EnrollStudent(self, request: academic_pb2.EnrollmentRequest, context) -> academic_pb2.EnrollmentResponse:  # type: ignore[override]
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details("EnrollStudent via gRPC not fully implemented yet.")
    return academic_pb2.EnrollmentResponse()


async def serve() -> None:
  server = grpc.aio.server()
  academic_pb2_grpc.add_AcademicServiceServicer_to_server(AcademicServiceGRPC(), server)
  listen_addr = f"[::]:{settings.academic_service_grpc_port}"
  server.add_insecure_port(listen_addr)
  await server.start()
  print(f"gRPC AcademicService listening on {listen_addr}")
  await server.wait_for_termination()


if __name__ == "__main__":
  asyncio.run(serve())


